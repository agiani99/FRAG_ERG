"""
Purchasable analogues of generated molecules (README 2.24): the link between grow.py output and the ECFP4 store.

  python analogues.py grow_6h0f_polar.csv --out grow_6h0f_polar_analogues.csv

Reads the `smiles` column of a grow.py result CSV, searches the purchasable ECFP4 store (ecfp_store.py, Tanimoto >= 0.85 by
default) and writes one row per (generated molecule, purchasable analogue) with CID, tier, vendors, SID link and bioactivity
summary. Prints the benchmark metric: the fraction of the generated molecules that have at least one purchasable analogue
(and at least one in the tight tier, and at least one exact match = Tanimoto 1.0).
grow.py is not modified by this script (it only reads the CSV it wrote).
"""
import paths
import argparse
import pandas as pd
import ecfp_store as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--smiles-col', default='smiles')
    ap.add_argument('--store', default='ecfp_purch')
    ap.add_argument('--idx', default=paths.pubchem_idx())
    ap.add_argument('--flags', default='purchasable_flags.npy')
    ap.add_argument('--vendors', default='vendor_map')
    ap.add_argument('--bio', default='bio_map')
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--top', type=int, default=3, help='analogues printed per molecule')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    qs = [(f'm{i}', s) for i, s in enumerate(d[a.smiles_col])]
    res = E.search(a, qs)
    out = a.out or a.csv.rsplit('.', 1)[0] + '_analogues.csv'
    res.to_csv(out, index=False)
    n = len(qs)
    have = res['query'].nunique() if len(res) else 0
    tight = res[res.tier == 'tight']['query'].nunique() if len(res) else 0
    exact = res[res.tanimoto >= 0.999]['query'].nunique() if len(res) else 0
    print(f'\n{n} generated molecules: {have} ({have / max(n, 1):.0%}) have a purchasable analogue >= {a.threshold}; '
          f'{tight} with a tight-tier one; {exact} with an exact ECFP4 match (Tanimoto 1.0)')
    pd.set_option('display.width', 250, 'display.max_colwidth', 70)
    cols = [c for c in ('cid', 'tanimoto', 'tier', 'n_vendors', 'bio_n_active', 'bio_n_tested', 'bio_best_nM', 'smiles') if c in res]
    for q, g in res.groupby('query', sort=False):
        print(f'\n{q}: {d[a.smiles_col].iloc[int(q[1:])]}')
        print(g.head(a.top)[cols].to_string(index=False))
    print(f'\nwritten {out}')


if __name__ == '__main__':
    main()
