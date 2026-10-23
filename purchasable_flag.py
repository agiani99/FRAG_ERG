"""
Purchasable flag for PubChem compounds from SID-Map.gz + the source table (see README 2.20).

  curl -O https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/Extras/SID-Map.gz            # 3.3 GB, user-run
  python purchasable_flag.py scan SID-Map.gz --out purchasable                         # stream the file, one core
  python purchasable_flag.py join --idx ../pubchem_idx --out purchasable               # flag per row of the ErG / known index

scan : SID-Map.gz has the columns SID, source name, registry id and (optional) standardized CID. Every row whose source is a
       vendor source and that carries a CID sets bits in flags[CID]:
         1 = tight   (named aggregators / screening-library suppliers, TIGHT below)
         2 = medium  (tight + every 'Chemical Vendors' source with >= 100,000 live substances)
         4 = zinc    (ZINC: catalogue aggregator, SECOND-HAND evidence; not part of medium)
       Output: <out>_flags.npy (uint8, index = CID), <out>_sources.csv (per source: tier, rows, rows with CID),
       <out>_unmatched.csv (SID-Map source names that are not in the source table, by row count).
join : looks up every row id (CID) of the index shards -> <out>_rows.npy (uint8 [N rows], same bit meaning, global row order
       of pubchem_idx / known_idx).
Purchasable (user decision 2026-10-06) = flags & 2, plus ZINC-only (flags & 4), reported separately as second-hand.
"""
import paths
import argparse, json, os, time
from collections import Counter
import numpy as np
import pandas as pd

TIGHT = ['molport', 'mcule', 'enamine', 'chem-space.com database', 'emolecules', 'chembridge', 'chemdiv',
         'vitas-m laboratory', 'life chemicals', 'interbioscreen', 'specs', 'asinex']
TIGHT_PREFIX = ['molmall']
MEDIUM_MIN = 100_000


def norm(s):
    return s.strip().lower()


def source_tiers(csv='pubchem_sources.csv'):
    df = pd.read_csv(csv)
    tier = {}
    for name, n, cat in zip(df['Source Name'], df['n'] if 'n' in df else df.iloc[:, 1], df['Source Category'].fillna('')):
        k = norm(str(name))
        t = 0
        vendor = 'Chemical Vendors' in cat
        if k in TIGHT or any(k.startswith(p) for p in TIGHT_PREFIX):
            t = 1 | 2
        elif k == 'zinc':
            t = 4
        elif vendor and int(n) >= MEDIUM_MIN:
            t = 2
        tier[k] = t
    return tier


def scan(a):
    tier = source_tiers(a.sources)
    flags = np.zeros(a.max_cid + 1, dtype=np.uint8)
    rows, rows_cid, unmatched = Counter(), Counter(), Counter()
    t0 = time.time()
    n = 0
    rd = pd.read_csv(a.file, sep='\t', header=None, names=['sid', 'src', 'reg', 'cid'],
                     dtype={'sid': 'int64', 'src': 'string', 'reg': 'string', 'cid': 'float64'}, chunksize=a.chunk,
                     nrows=a.limit_rows, quoting=3, on_bad_lines='skip')
    for ch in rd:
        n += len(ch)  # all four columns are read: usecols fails on a chunk where no row has a CID
        src = ch['src'].str.strip().str.lower()
        vc = src.value_counts()
        for k, v in vc.items():
            if k in tier:
                if tier[k]:
                    rows[k] += int(v)
            else:
                unmatched[k] += int(v)
        t = src.map(tier).fillna(0).astype(np.uint8).to_numpy()
        cid = ch['cid'].to_numpy()
        ok = (t > 0) & ~np.isnan(cid)
        if ok.any():
            c = cid[ok].astype(np.int64)
            tt = t[ok]
            if c.max() >= len(flags):
                raise SystemExit(f'CID {c.max()} > --max-cid {a.max_cid}; rerun with a larger --max-cid')
            for bit in (1, 2, 4):
                sel = (tt & bit) > 0
                if sel.any():
                    flags[c[sel]] |= bit
            for k, v in src[ok].value_counts().items():
                rows_cid[k] += int(v)
        if (n // a.chunk) % 10 == 0:
            print(f'{n / 1e6:.0f}M rows, {(flags > 0).sum() / 1e6:.2f}M flagged CIDs, {(time.time() - t0) / 60:.1f} min', flush=True)
    np.save(a.out + '_flags.npy', flags)
    pd.DataFrame([(k, tier[k], rows[k], rows_cid[k]) for k in rows], columns=['source', 'tier_bits', 'rows', 'rows_with_cid']) \
        .sort_values('rows', ascending=False).to_csv(a.out + '_sources.csv', index=False)
    pd.DataFrame(unmatched.most_common(300), columns=['source_in_SID_Map', 'rows']).to_csv(a.out + '_unmatched.csv', index=False)
    print(f'done: {n} rows in {(time.time() - t0) / 60:.1f} min; CIDs tight {(flags & 1 > 0).sum()}, medium {(flags & 2 > 0).sum()}, '
          f'zinc {(flags & 4 > 0).sum()}, zinc-only {((flags & 4 > 0) & (flags & 2 == 0)).sum()}')


def join(a):
    flags = np.load(a.out + '_flags.npy')
    meta = json.load(open(os.path.join(a.idx, 'meta.json')))
    parts = []
    for s in meta['shards']:
        ids = pd.read_parquet(os.path.join(a.idx, f"{s['name']}.ids.parquet"), columns=['id']).id.to_numpy().astype(np.int64)
        f = np.zeros(len(ids), dtype=np.uint8)
        ok = ids < len(flags)
        f[ok] = flags[ids[ok]]
        parts.append(f)
        print(s['name'], len(ids), int((f > 0).sum()), flush=True)
    rows = np.concatenate(parts)
    np.save(a.out + '_rows.npy', rows)
    print(f'{len(rows)} rows; tight {(rows & 1 > 0).sum()}, medium {(rows & 2 > 0).sum()}, zinc-only {((rows & 4 > 0) & (rows & 2 == 0)).sum()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('scan')
    s.add_argument('file')
    s.add_argument('--out', default='purchasable')
    s.add_argument('--sources', default='pubchem_sources.csv')
    s.add_argument('--chunk', type=int, default=5_000_000)
    s.add_argument('--max-cid', type=int, default=400_000_000)
    s.add_argument('--limit-rows', type=int, default=None)
    j = sp.add_parser('join')
    j.add_argument('--idx', default=paths.pubchem_idx())
    j.add_argument('--out', default='purchasable')
    a = ap.parse_args()
    scan(a) if a.cmd == 'scan' else join(a)
