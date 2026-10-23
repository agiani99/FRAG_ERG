"""
Vendor annotation of PubChem compounds (README 2.22): which vendor sources list a CID, and through which SID.

  python vendor_annotate.py build SID-Map.gz --out vendor_map          # one pass over SID-Map.gz (~6 min, ~6 GB RAM), user-run
  python vendor_annotate.py lookup 2244 5090 --map vendor_map          # quick check of a few CIDs

build: keeps every SID-Map row whose source has the category 'Chemical Vendors' (all 544 sources, ZINC included), whose CID is
       flagged purchasable in purchasable_flags.npy (medium / tight / ZINC; --all-cids keeps every CID), one row per
       (CID, source) with the first SID seen. Output (sorted by CID): <out>_cid.npy int32, <out>_src.npy uint16,
       <out>_sid.npy int64, <out>_sources.csv (src id, name, tier bits, URL).
VendorMap.annotate(cids) -> per CID: n_vendors, vendors (tight sources first, then by catalogue size; ZINC marked '(ZINC,
second-hand)'), and the SID of the best source (PubChem page https://pubchem.ncbi.nlm.nih.gov/substance/<SID>).
Registry (catalogue) numbers are not stored; the SID page shows them. A listing is evidence of offer, not of stock.
"""
import argparse, os, time
import numpy as np
import pandas as pd

import purchasable_flag as P


def build(a):
    src_df = pd.read_csv(a.sources)
    tier = P.source_tiers(a.sources)
    cat = {P.norm(str(n)): str(c) for n, c in zip(src_df['Source Name'], src_df['Source Category'].fillna(''))}
    names = {P.norm(str(n)): str(n).strip() for n in src_df['Source Name']}
    size = {P.norm(str(n)): int(v) for n, v in zip(src_df['Source Name'], src_df['n'])}
    vend = [k for k, c in cat.items() if 'Chemical Vendors' in c or k == 'zinc']
    vend.sort(key=lambda k: (-(tier.get(k, 0) & 1), -size[k]))      # tight first, then by catalogue size
    sid_of = {k: i for i, k in enumerate(vend)}
    flags = np.load(a.flags) if not a.all_cids else None
    cids, srcs, sids = [], [], []
    t0, n = time.time(), 0
    rd = pd.read_csv(a.file, sep='\t', header=None, names=['sid', 'src', 'reg', 'cid'],
                     dtype={'sid': 'int64', 'src': 'string', 'reg': 'string', 'cid': 'float64'}, chunksize=a.chunk,
                     nrows=a.limit_rows, quoting=3, on_bad_lines='skip')
    for ch in rd:
        n += len(ch)
        s = ch['src'].str.strip().str.lower().map(sid_of)
        cid = ch['cid'].to_numpy()
        ok = s.notna().to_numpy() & ~np.isnan(cid)
        if not ok.any():
            continue
        c = cid[ok].astype(np.int64)
        if flags is not None:
            keep = (c < len(flags)) & (flags[np.minimum(c, len(flags) - 1)] > 0)
            c = c[keep]
        else:
            keep = np.ones(len(c), bool)
        cids.append(c.astype(np.int32))
        srcs.append(s.to_numpy()[ok][keep].astype(np.uint16))
        sids.append(ch['sid'].to_numpy()[ok][keep])
        if (n // a.chunk) % 10 == 0:
            print(f'{n / 1e6:.0f}M rows, {sum(len(x) for x in cids) / 1e6:.1f}M kept, {(time.time() - t0) / 60:.1f} min', flush=True)
    cid = np.concatenate(cids); src = np.concatenate(srcs); sid = np.concatenate(sids)
    key = cid.astype(np.int64) * 1024 + src                          # 544 sources < 1024
    _, first = np.unique(key, return_index=True)                      # sorted by (CID, source rank)
    cid, src, sid = cid[first], src[first], sid[first]
    np.save(a.out + '_cid.npy', cid); np.save(a.out + '_src.npy', src); np.save(a.out + '_sid.npy', sid)
    url = dict(zip([P.norm(str(n)) for n in src_df['Source Name']], src_df['Source URL'].fillna('')))
    pd.DataFrame([(i, names[k], tier.get(k, 0), url.get(k, '')) for k, i in sid_of.items()],
                 columns=['src', 'name', 'tier_bits', 'url']).to_csv(a.out + '_sources.csv', index=False)
    print(f'done: {n} rows, {len(cid)} (CID, source) pairs, {len(np.unique(cid))} CIDs, {(time.time() - t0) / 60:.1f} min')


class VendorMap:
    def __init__(self, path='vendor_map'):
        self.cid = np.load(path + '_cid.npy', mmap_mode='r')
        self.src = np.load(path + '_src.npy', mmap_mode='r')
        self.sid = np.load(path + '_sid.npy', mmap_mode='r')
        self.sources = pd.read_csv(path + '_sources.csv')
        self.name = self.sources.name.tolist()
        self.tier = self.sources.tier_bits.tolist()

    def annotate(self, cids, max_names=6):
        """dict cid -> (n_vendors, 'a; b; c', sid of the first source, n_tight)"""
        out = {}
        q = np.array(sorted(set(int(x) for x in cids)), dtype=np.int32)
        lo = np.searchsorted(self.cid, q, 'left')          # one vectorised lookup (a per-CID call on the memory map was ~0.5 s each)
        hi = np.searchsorted(self.cid, q, 'right')
        for c, l, h in zip(q.tolist(), lo.tolist(), hi.tolist()):
            if h == l:
                out[c] = (0, '', 0, 0)
                continue
            s = np.asarray(self.src[l:h])                    # ascending = tight first, then by size
            lab = [self.name[i] + (' (second-hand)' if self.name[i] == 'ZINC' else '') for i in s]
            ntight = sum(1 for i in s if self.tier[i] & 1)
            out[c] = (h - l, '; '.join(lab[:max_names]) + (f'; +{len(lab) - max_names} more' if len(lab) > max_names else ''),
                      int(self.sid[l]), ntight)
        return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build')
    b.add_argument('file')
    b.add_argument('--out', default='vendor_map')
    b.add_argument('--sources', default='pubchem_sources.csv')
    b.add_argument('--flags', default='purchasable_flags.npy')
    b.add_argument('--all-cids', action='store_true')
    b.add_argument('--chunk', type=int, default=5_000_000)
    b.add_argument('--limit-rows', type=int, default=None)
    l = sp.add_parser('lookup')
    l.add_argument('cids', nargs='+', type=int)
    l.add_argument('--map', default='vendor_map')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a)
    else:
        for c, v in VendorMap(a.map).annotate(a.cids).items():
            print(c, v)
