"""How much does a library compress into unique BRICS building blocks (site-aware cuts)?

usage: python compress.py [molecules_per_shard=30000] [workers=19] [options]

  --neutralize          neutralise every molecule first (RDKit Uncharger: protonated amines, carboxylates,
                        ... -> neutral; permanent charges such as quaternary N and nitro are kept)
  --glob PATTERN        input parquet shards   (default $FRAGERG_DATA/pubchem_idx/shard_*.ids.parquet)
  --col NAME            SMILES column          (default smiles)
  --out FILE            output block counts    (default fragment_counts_pubchem_sample.parquet;
                        use e.g. fragment_counts_pubchem_neutral.parquet for a neutral run so the
                        charged library is not overwritten)

Then:  python grow.py --library <that parquet> --smiles "<reference in the same protonation convention>"
"""
import paths
import argparse, glob, collections, time, multiprocessing as mp
import numpy as np, pandas as pd
from rdkit import Chem
import frag_erg as F

_UNCH = None


def work(args):
    smis, neutralize = args
    global _UNCH
    if neutralize and _UNCH is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _UNCH = rdMolStandardize.Uncharger()
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append(None)
            continue
        if neutralize:
            try:
                m = _UNCH.uncharge(m)
            except Exception:
                pass
        try:
            out.append(F.fragment_keys(m))
        except Exception:
            out.append(None)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('per_shard', nargs='?', type=int, default=30000)
    ap.add_argument('workers', nargs='?', type=int, default=19)
    ap.add_argument('--neutralize', action='store_true')
    ap.add_argument('--glob', default=paths.pubchem_idx() + '/shard_*.ids.parquet')
    ap.add_argument('--col', default='smiles')
    ap.add_argument('--out', default='fragment_counts_pubchem_sample.parquet')
    a = ap.parse_args()

    shards = sorted(glob.glob(a.glob))
    smis = []
    for k, p in enumerate(shards):
        s = pd.read_parquet(p, columns=[a.col])[a.col]
        smis += s.sample(min(a.per_shard, len(s)), random_state=k).tolist()
    np.random.default_rng(0).shuffle(smis)
    print(len(smis), 'molecules sampled from', len(shards), 'shards',
          '(neutralized)' if a.neutralize else '', flush=True)
    chunks = [(smis[i:i + 2000], a.neutralize) for i in range(0, len(smis), 2000)]
    cnt = collections.Counter()
    nmol = nfrag = nfail = 0
    t = time.time()
    nxt = [100_000, 250_000, 500_000, 750_000, 1_000_000]
    curve = []
    with mp.Pool(a.workers) as pool:
        for res in pool.imap(work, chunks):
            for r in res:
                if r is None:
                    nfail += 1
                    continue
                nmol += 1
                nfrag += len(r)
                cnt.update(r)
            while nxt and nmol >= nxt[0]:
                curve.append((nxt[0], len(cnt)))
                nxt.pop(0)
    curve.append((nmol, len(cnt)))
    print(f'{nmol} mols ({nfail} failed) in {time.time() - t:.0f}s; '
          f'{nfrag} fragment instances ({nfrag / nmol:.2f}/mol)')
    print('growth (molecules -> unique fragments):', curve)
    u = len(cnt)
    print(f'unique fragments {u} -> {nmol / u:.2f} molecules per unique block, '
          f'{nfrag / u:.1f} block instances per unique block')
    c = np.array(sorted(cnt.values(), reverse=True))
    print(f'singletons {np.mean(c == 1):.1%} of unique; they carry {c[c == 1].sum() / nfrag:.1%} of instances')
    cs = np.cumsum(c) / nfrag
    for q in (0.5, 0.8, 0.9, 0.95, 0.99):
        print(f'  top {np.searchsorted(cs, q) + 1} fragments cover {q:.0%} of all fragment instances')
    print('top 15:')
    for k, n in cnt.most_common(15):
        print('  ', n, k)
    pd.Series(cnt).sort_values(ascending=False).rename('count').to_frame().to_parquet(a.out)
    print('written', a.out)
