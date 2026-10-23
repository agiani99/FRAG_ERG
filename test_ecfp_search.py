"""Exactness test of ecfp_store.search on the pilot store (brute force over all stored fingerprints)."""
import argparse, os, numpy as np, pandas as pd
import paths
import ecfp_store as E

if __name__ == '__main__':
    st = 'ecfp_pilot'
    idx = paths.pubchem_idx()
    if not os.path.exists(os.path.join(st, 'shard_00000.rows.npy')):      # self-contained: build the 300k-row pilot store (~30 s)
        E.build(argparse.Namespace(idx=idx, flags='purchasable_rows.npy', out=st, mask=6, workers=8, chunk=20000,
                                   limit_rows=300000, max_shards=1))
    fp = np.load(f'{st}/shard_00000.fp.npy'); rows = np.load(f'{st}/shard_00000.rows.npy')
    ids = pd.read_parquet(os.path.join(idx, 'shard_00000.ids.parquet'))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), 25, replace=False)
    qs = [(f'self{i}', ids.smiles.iloc[int(rows[i])]) for i in pick]
    qs += [('imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'),
           ('aspirin', 'CC(=O)Oc1ccccc1C(=O)O'),
           ('piperidine_amide', 'COc1ccc(S(=O)(=O)N2CCC(C(=O)Nc3ccccc3)CC2)cc1')]
    a = argparse.Namespace(store=st, idx=idx, flags='purchasable_flags.npy', threshold=0.85)
    df = E.search(a, qs)
    bits = np.unpackbits(fp, axis=1).astype(np.float32)
    bad = 0
    for name, smi in qs:
        b = np.unpackbits(E.ecfp_bits(smi)).astype(np.float32)
        inter = bits @ b
        t = inter / (bits.sum(1) + b.sum() - inter)
        truth = set(rows[np.nonzero(t >= 0.85 - 1e-6)[0]].tolist())
        got = set(df[df['query'] == name].row.tolist())
        bad += truth != got
        print(f'{name:18s} truth {len(truth):3d} got {len(got):3d}', '' if truth == got else 'MISMATCH')
    print('mismatching queries:', bad)
    print(df[df['query'] == 'piperidine_amide'][['cid', 'tanimoto', 'tier', 'smiles']].head(8).to_string(index=False))
