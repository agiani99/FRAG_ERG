"""
Bioactivity annotation of purchasable PubChem compounds from bioactivities.tsv.gz (README 2.23).

  curl -O https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/bioactivities.tsv.gz        # 3.04 GB, user-run
  python bio_annotate.py build bioactivities.tsv.gz --out bio_map                          # one pass, user-run
  python bio_annotate.py lookup 2244 5090 --map bio_map

File layout (tab separated, header line): AID, SID, SID Group, CID, Activity Outcome, Activity Name, Activity Qualifier,
Activity Value, Activity Unit, Protein Accession, Gene ID, Target TaxID, PMID. One row = one test result of one substance.

build: keeps rows whose CID is flagged purchasable (purchasable_flags.npy > 0) and writes
   <out>_summary.parquet  one row per CID: n_tested (all test results), n_active (outcome Active or Probe), n_inactive,
                          n_gene_active (distinct Gene IDs among active results), best_nM / best_gene (lowest potency value among
                          active results with a Gene ID, name IC50 / Ki / Kd / EC50 / AC50 / Potency, qualifier not > or >=)
   <out>_targets.parquet  one row per (CID, Gene ID) with at least one active result: n_active, best_nM (NaN if no potency)
A high n_active / n_tested ratio across many unrelated assays points at promiscuity or assay interference, not at a
selective repurposing lead; the counts are reported side by side. Gene IDs are NCBI Gene numbers (no symbols offline).
"""
import argparse, os, time
import numpy as np
import pandas as pd

UNIT_NM = {'nm': 1.0, 'um': 1e3, 'µm': 1e3, 'μm': 1e3, 'mm': 1e6, 'pm': 1e-3, 'm': 1e9, 'fm': 1e-6}
POT = {'ic50', 'ki', 'kd', 'ec50', 'ac50', 'potency'}


def build(a):
    flags = np.load(a.flags, mmap_mode='r')
    n_c = a.max_cid + 1
    tested = np.zeros(n_c, dtype=np.uint32)
    active = np.zeros(n_c, dtype=np.uint32)
    inactive = np.zeros(n_c, dtype=np.uint32)
    parts = []
    t0, n, kept = time.time(), 0, 0
    rd = pd.read_csv(a.file, sep='\t', header=0, usecols=[3, 4, 5, 6, 7, 8, 10], dtype=str, chunksize=a.chunk,
                     nrows=a.limit_rows, quoting=3, on_bad_lines='skip')
    for ch in rd:
        ch.columns = ['cid', 'outcome', 'name', 'qual', 'value', 'unit', 'gene']
        n += len(ch)
        cid = pd.to_numeric(ch['cid'], errors='coerce').to_numpy()
        ok = ~np.isnan(cid)
        c = np.where(ok, cid, 0).astype(np.int64)
        if c.max() >= n_c:
            raise SystemExit(f'CID {c.max()} > --max-cid {a.max_cid}')
        ok &= flags[np.minimum(c, len(flags) - 1)] > 0
        ch, c = ch[ok], c[ok]
        if not len(ch):
            continue
        kept += len(ch)
        oc = ch['outcome'].fillna('').str.lower().to_numpy()
        act = np.isin(oc, ['active', 'probe'])
        ina = oc == 'inactive'
        for arr, m in ((tested, np.ones(len(c), bool)), (active, act), (inactive, ina)):
            u, k = np.unique(c[m], return_counts=True)
            arr[u] += k.astype(np.uint32)
        sel = act & ch['gene'].notna().to_numpy()
        sub = ch[sel].copy()
        if len(sub):
            sub['cid'] = c[sel]
            sub['gene'] = pd.to_numeric(sub['gene'].str.split(r'[;,|]').str[0], errors='coerce')
            sub = sub[sub['gene'].notna()]
        if len(sub):
            nm = np.full(len(sub), np.nan)
            isp = sub['name'].fillna('').str.lower().isin(POT).to_numpy() & ~sub['qual'].fillna('').isin(['>', '>=']).to_numpy()
            val = pd.to_numeric(sub['value'], errors='coerce').to_numpy()
            fac = sub['unit'].fillna('').str.lower().map(UNIT_NM).to_numpy(dtype=float)
            good = isp & ~np.isnan(val) & ~np.isnan(fac) & (val > 0)
            nm[good] = val[good] * fac[good]
            g = pd.DataFrame({'cid': sub['cid'].to_numpy(), 'gene': sub['gene'].to_numpy().astype(np.int64), 'nM': nm, 'n': 1})
            parts.append(g.groupby(['cid', 'gene'], as_index=False).agg(n=('n', 'sum'), nM=('nM', 'min')))
        if (n // a.chunk) % 10 == 0:
            print(f'{n / 1e6:.0f}M rows, {kept / 1e6:.1f}M flagged, {(time.time() - t0) / 60:.1f} min', flush=True)
    if parts:
        tg = pd.concat(parts).groupby(['cid', 'gene'], as_index=False).agg(n_active=('n', 'sum'), best_nM=('nM', 'min'))
    else:
        tg = pd.DataFrame({'cid': np.zeros(0, np.int64), 'gene': np.zeros(0, np.int64), 'n_active': np.zeros(0, int),
                           'best_nM': np.zeros(0)})
    tg = tg.sort_values(['cid', 'gene'])
    tg['cid'] = tg['cid'].astype(np.int32)
    tg.to_parquet(a.out + '_targets.parquet', index=False)
    cids = np.nonzero(tested)[0]
    s = pd.DataFrame({'cid': cids.astype(np.int32), 'n_tested': tested[cids], 'n_active': active[cids], 'n_inactive': inactive[cids]})
    ng = tg.groupby('cid').size()
    s['n_gene_active'] = s.cid.map(ng).fillna(0).astype(np.int32)
    pot = tg[tg.best_nM.notna()].sort_values('best_nM').drop_duplicates('cid').set_index('cid')
    s['best_nM'] = s.cid.map(pot.best_nM)
    s['best_gene'] = s.cid.map(pot.gene).fillna(0).astype(np.int64)
    s.to_parquet(a.out + '_summary.parquet', index=False)
    print(f'done: {n} rows, {kept} for flagged CIDs, {len(s)} CIDs with data, {int((s.n_active > 0).sum())} with an active result, '
          f'{len(tg)} (CID, gene) pairs, {(time.time() - t0) / 60:.1f} min')


class BioMap:
    def __init__(self, path='bio_map', cids=None):
        """cids: load only these CIDs (parquet row-group filter, seconds instead of minutes for the 60M-row summary)."""
        flt = [('cid', 'in', sorted(set(int(c) for c in cids)))] if cids is not None else None
        self.s = pd.read_parquet(path + '_summary.parquet', filters=flt).set_index('cid')
        self.t = pd.read_parquet(path + '_targets.parquet', filters=flt)

    def annotate(self, cids):
        """DataFrame indexed by CID: n_tested, n_active, n_inactive, n_gene_active, best_nM, best_gene (NaN when never tested)."""
        return self.s.reindex(sorted(set(int(c) for c in cids)))

    def targets(self, cid, top=10):
        t = self.t[self.t.cid == cid].sort_values(['best_nM', 'n_active'], ascending=[True, False])
        return t.head(top)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build')
    b.add_argument('file')
    b.add_argument('--out', default='bio_map')
    b.add_argument('--flags', default='purchasable_flags.npy')
    b.add_argument('--max-cid', type=int, default=200_000_000)
    b.add_argument('--chunk', type=int, default=5_000_000)
    b.add_argument('--limit-rows', type=int, default=None)
    l = sp.add_parser('lookup')
    l.add_argument('cids', nargs='+', type=int)
    l.add_argument('--map', default='bio_map')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a)
    else:
        m = BioMap(a.map)
        print(m.annotate(a.cids).to_string())
        for c in a.cids:
            print(c)
            print(m.targets(c).to_string(index=False))
