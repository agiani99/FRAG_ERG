"""
ECFP4 store of the purchasable PubChem rows and the analogue search (Tanimoto >= 0.85 by default). README 2.21.

  # D1 build (long, resumable per shard; user-run)
  python ecfp_store.py build --idx ../pubchem_idx --flags purchasable_rows.npy --out ecfp_purch --workers 19
  # pilot: first 300k rows of the first shard
  python ecfp_store.py build --idx ../pubchem_idx --flags purchasable_rows.npy --out ecfp_pilot --workers 8 --limit-rows 300000 --max-shards 1

  # D2 search: SMILES file (one per line, optional name after a space) or --smiles "..." (repeatable)
  python ecfp_store.py search --store ecfp_purch --idx ../pubchem_idx --smiles-file cands.smi --out analogues.csv

Fingerprint = Morgan radius 2, 2048 bits (ECFP4, bits not counts), computed on the NEUTRALISED largest component (the index
stores charged forms; the same function is applied to the stored rows and to the queries, so the two sides match).
Store per shard (rows kept = flag & --mask, default 6 = medium + ZINC-only), sorted by popcount:
    <shard>.fp.npy   uint8  [n, 256]   packed bits          <shard>.pc.npy    uint16 [n]  popcount (ascending)
    <shard>.rows.npy int32  [n]        global row in pubchem_idx
Search: for each query only the popcount window [thr*a, a/thr] can reach the threshold (Tanimoto <= min/max of popcounts);
numba kernel over uint64 words; each shard is read once per batch of queries. Hits carry (with --vendors: n_vendors, n_tight, vendor names, SID and its PubChem URL) CID (= the index id), SMILES as
stored, the tier label (tight / medium / zinc-only = second-hand vendor evidence) and the Tanimoto.
"""
import paths
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

import known as K

RDLogger.DisableLog('rdApp.*')
NB = 2048
_GEN = None
_UN = None


def _init():
    global _GEN, _UN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NB)
        _UN = rdMolStandardize.Uncharger()


def ecfp_bits(smi):
    """Packed ECFP4 (uint8 [256]) of the neutralised largest component of a SMILES, or None."""
    _init()
    m = Chem.MolFromSmiles(max(smi.split('.'), key=len))
    if m is None:
        return None
    try:
        m = _UN.uncharge(K.repair_charges(m))
        return np.packbits(_GEN.GetFingerprintAsNumPy(m).astype(np.uint8))
    except Exception:
        return None


def _chunk(smis):
    out = np.zeros((len(smis), NB // 8), dtype=np.uint8)
    ok = np.zeros(len(smis), dtype=bool)
    for i, s in enumerate(smis):
        b = ecfp_bits(s)
        if b is not None:
            out[i] = b
            ok[i] = True
    return out, ok


POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)


def popcount(fp):
    return POP8[fp].sum(axis=1).astype(np.uint16)


def build(a):
    flags = np.load(a.flags)
    meta = json.load(open(os.path.join(a.idx, 'meta.json')))
    offsets = np.cumsum([0] + [s['n'] for s in meta['shards']])
    shards = meta['shards'][:a.max_shards] if a.max_shards else meta['shards']
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tot = 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for si, s in enumerate(shards):
            base = os.path.join(a.out, s['name'])
            if os.path.exists(base + '.rows.npy'):
                print(f"[{s['name']}] done already - skipped", flush=True)
                continue
            t1 = time.time()
            smi = pd.read_parquet(os.path.join(a.idx, f"{s['name']}.ids.parquet"), columns=['smiles']).smiles.tolist()
            smi_all = smi if not a.limit_rows else smi[:a.limit_rows]
            rows = np.arange(offsets[si], offsets[si] + len(smi), dtype=np.int64)
            if a.limit_rows:
                smi, rows = smi[:a.limit_rows], rows[:a.limit_rows]
            keep = np.nonzero((flags[rows] & a.mask) > 0)[0]
            smi = [smi[i] for i in keep]
            rows = rows[keep]
            res = pool.map(_chunk, [smi[i:i + a.chunk] for i in range(0, len(smi), a.chunk)])
            fp = np.concatenate([r[0] for r in res]) if res else np.zeros((0, NB // 8), np.uint8)
            ok = np.concatenate([r[1] for r in res]) if res else np.zeros(0, bool)
            fp, rows = fp[ok], rows[ok]
            pc = popcount(fp)
            o = np.argsort(pc, kind='stable')
            for ext, arr in (('fp', fp[o]), ('pc', pc[o]), ('rows', rows[o].astype(np.int32))):   # rows last: marks the shard as done
                np.save(f'{base}.{ext}.tmp.npy', arr)
                os.replace(f'{base}.{ext}.tmp.npy', f'{base}.{ext}.npy')
            tot += len(fp)
            print(f"[{s['name']}] kept {len(keep)} of {len(smi_all)}, fingerprints {len(fp)} "
                  f"({len(keep) - int(ok.sum())} failed), {time.time() - t1:.0f}s; total {tot}; {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    json.dump({'mask': a.mask, 'nbits': NB, 'shards': [s['name'] for s in shards], 'idx': a.idx},
              open(os.path.join(a.out, 'info.json'), 'w'))


# ------------------------------------------------------------------ search
def _kernel():
    import numba as nb

    @nb.njit(inline='always')
    def pop64(x):
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

    @nb.njit(parallel=True, cache=True)
    def sims(fp64, pc, lo, hi, q64, qpc):
        out = np.empty(hi - lo, dtype=np.float32)
        for k in nb.prange(hi - lo):
            i = lo + k
            inter = np.uint64(0)
            for w in range(fp64.shape[1]):
                inter += pop64(fp64[i, w] & q64[w])
            u = np.float32(pc[i]) + np.float32(qpc) - np.float32(inter)
            out[k] = np.float32(inter) / u if u > 0 else np.float32(0)
        return out
    return sims


TIER = {1: 'tight', 2: 'medium', 4: 'zinc-only (second-hand)'}


def tier_label(f):
    return 'tight' if f & 1 else 'medium' if f & 2 else 'zinc-only (second-hand)' if f & 4 else 'none'


def search(a, queries=None):
    """queries: list of (name, smiles). Returns a DataFrame of analogues (also written to --out)."""
    sims = _kernel()
    info = json.load(open(os.path.join(a.store, 'info.json')))
    flags = np.load(a.flags)
    q = []
    for name, smi in queries:
        b = ecfp_bits(smi)
        if b is None:
            print(f'query {name}: could not be parsed', file=sys.stderr)
            continue
        q.append((name, smi, b))
    qpc = [int(POP8[b].sum()) for _, _, b in q]
    hits = []
    t0 = time.time()
    meta = json.load(open(os.path.join(a.idx, 'meta.json')))
    offsets = dict(zip([s['name'] for s in meta['shards']], np.cumsum([0] + [s['n'] for s in meta['shards']])))
    for sname in info['shards']:
        base = os.path.join(a.store, sname)
        if not os.path.exists(base + '.rows.npy'):
            continue
        fp = np.load(base + '.fp.npy', mmap_mode='r')
        pc = np.load(base + '.pc.npy')
        rows = np.load(base + '.rows.npy')
        fp64 = fp.view(np.uint64) if fp.flags['C_CONTIGUOUS'] else np.ascontiguousarray(fp).view(np.uint64)
        pcs = np.ascontiguousarray(pc)
        found = []
        for j, (name, smi, b) in enumerate(q):
            lo = int(np.searchsorted(pcs, int(np.ceil(a.threshold * qpc[j])), 'left'))
            hi = int(np.searchsorted(pcs, int(np.floor(qpc[j] / a.threshold)), 'right'))
            if hi <= lo:
                continue
            s = sims(fp64, pcs, lo, hi, b.view(np.uint64), qpc[j])
            for k in np.nonzero(s >= a.threshold - 1e-6)[0]:
                found.append((j, int(lo + k), float(s[k])))
        if found:
            ids = pd.read_parquet(os.path.join(a.idx, f'{sname}.ids.parquet'))
            for j, i, s in found:
                g = int(rows[i])
                li = g - int(offsets[sname])
                hits.append(dict(query=q[j][0], query_smiles=q[j][1], cid=int(ids.id.iloc[li]), smiles=ids.smiles.iloc[li],
                                 tanimoto=round(s, 3), tier=tier_label(int(flags[int(ids.id.iloc[li])])), row=g))
    df = pd.DataFrame(hits, columns=['query', 'query_smiles', 'cid', 'smiles', 'tanimoto', 'tier', 'row'])
    if len(df):
        # the index holds several protonation forms per CID: one row per (query, CID); salts / mixtures are marked
        df['components'] = df.smiles.str.count(r'\.') + 1
        df = df.sort_values(['query', 'tanimoto', 'components'], ascending=[True, False, True]).drop_duplicates(['query', 'cid'])
    if len(df) and getattr(a, 'vendors', None):
        import vendor_annotate as V
        ann = V.VendorMap(a.vendors).annotate(df.cid.tolist())
        df['n_vendors'] = [ann[c][0] for c in df.cid]
        df['n_tight'] = [ann[c][3] for c in df.cid]
        df['vendors'] = [ann[c][1] for c in df.cid]
        df['sid'] = [ann[c][2] for c in df.cid]
        df['sid_url'] = [f'https://pubchem.ncbi.nlm.nih.gov/substance/{ann[c][2]}' if ann[c][2] else '' for c in df.cid]
    if len(df) and getattr(a, 'bio', None):
        import bio_annotate as B
        ba = B.BioMap(a.bio, df.cid.tolist()).annotate(df.cid.tolist())
        for col in ('n_tested', 'n_active', 'n_gene_active', 'best_nM', 'best_gene'):
            df['bio_' + col] = df.cid.map(ba[col])
    print(f'{len(q)} queries, {len(df)} analogues >= {a.threshold} in {time.time() - t0:.1f}s; '
          f'queries with a hit: {df["query"].nunique() if len(df) else 0}', flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build')
    b.add_argument('--idx', default=paths.pubchem_idx())
    b.add_argument('--flags', default='purchasable_rows.npy')
    b.add_argument('--out', default='ecfp_purch')
    b.add_argument('--mask', type=int, default=6)
    b.add_argument('--workers', type=int, default=19)
    b.add_argument('--chunk', type=int, default=20000)
    b.add_argument('--limit-rows', type=int, default=None)
    b.add_argument('--max-shards', type=int, default=None)
    s = sp.add_parser('search')
    s.add_argument('--store', default='ecfp_purch')
    s.add_argument('--idx', default=paths.pubchem_idx())
    s.add_argument('--flags', default='purchasable_flags.npy', help='by CID (scan output), for the tier label')
    s.add_argument('--vendors', default=None, help='vendor_map prefix (vendor_annotate.py build): adds vendor columns')
    s.add_argument('--bio', default=None, help='bio_map prefix (bio_annotate.py build): adds bioactivity columns')
    s.add_argument('--smiles', action='append', default=[])
    s.add_argument('--smiles-file')
    s.add_argument('--threshold', type=float, default=0.85)
    s.add_argument('--out', default='analogues.csv')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a)
    else:
        qs = [(f'q{i}', x) for i, x in enumerate(a.smiles)]
        if a.smiles_file:
            for i, l in enumerate(open(a.smiles_file)):
                p = l.split()
                if p:
                    qs.append((p[1] if len(p) > 1 else f'f{i}', p[0]))
        df = search(a, qs)
        df.to_csv(a.out, index=False)
        print(df.drop(columns=['query_smiles']).head(30).to_string(index=False))


if __name__ == '__main__':
    main()
