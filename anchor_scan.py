"""
Anchor-first scan: every purchasable molecule that contains the anchor of the reference (README 2.25).

  python anchor_scan.py --smiles "<reference>" --anchor-frag 1 --workers 19 --out hits_6h0f.parquet
  python retrieve.py --smiles "<reference>" --hits hits_6h0f.parquet --complex ... --anchor-frag 1 ...   # rank + pocket score

The anchor is chosen as in grow.py (--anchor-n / --anchor-frag; whole reference if it has no BRICS bond) or given as SMARTS
(--anchor-smarts). The scan reads the SMILES of the rows kept in the ECFP store (ecfp_purch/<shard>.rows.npy = medium +
ZINC-only purchasable rows, 66.7M), parses the largest component and tests the SMARTS (exact, no fingerprint screen) in a
worker pool. Heavy-atom window [ref - 8, ref + 12] by default (--min-heavy / --max-heavy) skips the parse of the match for
molecules of the wrong size. Matching is on the stored (charged) form: the query atoms carry no charge constraint.
--prescreen F (default 0 = off) first drops rows that have fewer than F x (ECFP4 bits of the free anchor) of those bits: a
FAST BUT INEXACT screen, measured in README 2.25 (recall 90-99 % keeping 5-40 % of the rows); use it only for repeat scans.
"""
import paths
import argparse, json, os, sys, time, warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
RDLogger.DisableLog('rdApp.*')
_Q = None
_LO = _HI = 0


def _init(smarts, lo, hi):
    global _Q, _LO, _HI
    _Q = Chem.MolFromSmarts(smarts)
    _LO, _HI = lo, hi


def _scan(smis):
    out = []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(max(s.split('.'), key=len))
        if m is None:
            continue
        n = m.GetNumHeavyAtoms()
        if n < _LO or n > _HI:
            continue
        if m.HasSubstructMatch(_Q):
            out.append(i)
    return out


def anchor_prescreen_bits(smarts_mol_smiles):
    from rdkit.Chem import rdFingerprintGenerator
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.nonzero(g.GetFingerprintAsNumPy(Chem.MolFromSmiles(smarts_mol_smiles)))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smiles', required=True, help='reference ligand')
    ap.add_argument('--anchor-n', type=int, default=1)
    ap.add_argument('--anchor-frag', type=int, default=None)
    ap.add_argument('--anchor-smarts', default=None)
    ap.add_argument('--store', default=os.path.join(HERE, 'ecfp_purch'))
    ap.add_argument('--idx', default=paths.pubchem_idx())
    ap.add_argument('--workers', type=int, default=19)
    ap.add_argument('--min-heavy', type=int, default=None)
    ap.add_argument('--max-heavy', type=int, default=None)
    ap.add_argument('--prescreen', type=float, default=0.0)
    ap.add_argument('--shards', type=int, default=None, help='first N shards only (test)')
    ap.add_argument('--chunk', type=int, default=20000)
    ap.add_argument('--out', default='hits.parquet')
    a = ap.parse_args()
    import erg_v2 as E
    import grow as G
    import retrieve as R
    ref = Chem.MolFromSmiles(a.smiles)
    nref = ref.GetNumHeavyAtoms()
    lo = a.min_heavy if a.min_heavy is not None else max(nref - 8, 1)
    hi = a.max_heavy if a.max_heavy is not None else nref + 12
    if a.anchor_smarts:
        smarts, free = a.anchor_smarts, None
    else:
        atoms = R.anchor_atoms(ref, G.Target(E.erg_raw(ref)), a)
        q, _ = R.anchor_query(ref, atoms)
        smarts = Chem.MolToSmarts(q)
        free = Chem.MolFragmentToSmiles(ref, atomsToUse=sorted(atoms))
    print(f'anchor SMARTS {smarts}; heavy atoms {lo}-{hi}', flush=True)
    sel_bits = None
    if a.prescreen > 0 and free:
        bits = anchor_prescreen_bits(free)
        thr = int(np.ceil(a.prescreen * len(bits)))
        print(f'prescreen: >= {thr} of the {len(bits)} ECFP4 bits of the free anchor {free} (inexact)')
        sel_bits = (bits // 8, (0x80 >> (bits % 8)).astype(np.uint8), thr)
    info = json.load(open(os.path.join(a.store, 'info.json')))
    meta = json.load(open(os.path.join(a.idx, 'meta.json')))
    offs = dict(zip([s['name'] for s in meta['shards']], np.cumsum([0] + [s['n'] for s in meta['shards']])))
    shards = info['shards'][:a.shards] if a.shards else info['shards']
    t0 = time.time()
    hits, total, parsed = [], 0, 0
    with mp.Pool(a.workers, initializer=_init, initargs=(smarts, lo, hi)) as pool:
        for sname in shards:
            rows = np.load(os.path.join(a.store, sname + '.rows.npy'))
            loc = rows.astype(np.int64) - int(offs[sname])
            if sel_bits is not None:
                fp = np.load(os.path.join(a.store, sname + '.fp.npy'), mmap_mode='r')
                cnt = np.zeros(len(rows), dtype=np.int16)
                for byte, mask in zip(sel_bits[0], sel_bits[1]):
                    cnt += ((np.asarray(fp[:, int(byte)]) & mask) > 0)
                keep = cnt >= sel_bits[2]
                rows, loc = rows[keep], loc[keep]
            ids = pd.read_parquet(os.path.join(a.idx, sname + '.ids.parquet'))
            smi = ids.smiles.to_numpy()[loc]
            cid = ids.id.to_numpy()[loc]
            total += len(rows)
            res = pool.map(_scan, [list(smi[i:i + a.chunk]) for i in range(0, len(smi), a.chunk)])
            hit = np.concatenate([np.array(r, dtype=np.int64) + i * a.chunk for i, r in enumerate(res)]) if res else np.zeros(0, int)
            hits.append(pd.DataFrame({'row': rows[hit].astype(np.int64), 'cid': cid[hit].astype(np.int64), 'smiles': smi[hit]}))
            print(f'[{sname}] {len(rows)} rows scanned, {len(hit)} hits; total hits {sum(len(h) for h in hits)}; '
                  f'{(time.time() - t0) / 60:.1f} min', flush=True)
    df = pd.concat(hits, ignore_index=True)
    df.attrs['anchor'] = smarts
    df.to_parquet(a.out, index=False)
    print(f'done: {total} rows scanned, {len(df)} molecules contain the anchor, {(time.time() - t0) / 60:.1f} min; written {a.out}')


if __name__ == '__main__':
    main()
