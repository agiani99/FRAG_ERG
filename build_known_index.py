"""
Build the PubChem 'known molecule' index (see known.py) from the SMILES stored in the ErG index.

  python build_known_index.py --idx ../pubchem_idx --out known_idx --workers 19
  python build_known_index.py --idx ../pubchem_idx --out known_pilot --limit-rows 300000 --max-shards 1   # quick pilot

Per shard (resumable: finished shards are skipped): read ids.parquet, InChIKey first block of every SMILES in worker
processes (about 0.6 ms per molecule per worker), write <shard>.keys.npy (uint64, 0 = SMILES that failed). When all shards
are done they are merged: valid keys sorted, with the global row numbers, into keys.npy / rows.npy / info.json.
Estimated cost for the 131M-molecule index: ~65 min on 19 workers; disk: ~1.6 GB.
"""
import paths
import argparse, json, os, time
import multiprocessing as mp
import numpy as np
import pandas as pd

import known as K


def _chunk(smis):
    out = np.zeros(len(smis), dtype=np.uint64)
    for i, s in enumerate(smis):
        k = K.smiles_key(s)
        if k:
            out[i] = K.h64(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', default=paths.pubchem_idx())
    ap.add_argument('--out', default='known_idx')
    ap.add_argument('--workers', type=int, default=19)
    ap.add_argument('--limit-rows', type=int, default=None, help='rows per shard (pilot)')
    ap.add_argument('--max-shards', type=int, default=None)
    ap.add_argument('--chunk', type=int, default=20000)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.idx, 'meta.json')))
    shards = meta['shards'][:a.max_shards] if a.max_shards else meta['shards']
    offsets = np.cumsum([0] + [s['n'] for s in meta['shards']])
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for si, s in enumerate(shards):
            part = os.path.join(a.out, f"{s['name']}.keys.npy")
            if os.path.exists(part):
                print(f"[{s['name']}] done already - skipped", flush=True)
                continue
            t1 = time.time()
            smi = pd.read_parquet(os.path.join(a.idx, f"{s['name']}.ids.parquet"), columns=['smiles']).smiles.tolist()
            if a.limit_rows:
                smi = smi[:a.limit_rows]
            keys = np.concatenate(pool.map(_chunk, [smi[i:i + a.chunk] for i in range(0, len(smi), a.chunk)]))
            tmp = part + '.tmp.npy'
            np.save(tmp, keys)
            os.replace(tmp, part)
            print(f"[{s['name']}] {len(keys)} molecules, {int((keys == 0).sum())} failed, {time.time() - t1:.0f}s "
                  f"({len(keys) / (time.time() - t1):.0f} mol/s); elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    # merge
    allk, allr, nfail = [], [], 0
    for si, s in enumerate(shards):
        k = np.load(os.path.join(a.out, f"{s['name']}.keys.npy"))
        ok = k != 0
        nfail += int((~ok).sum())
        allk.append(k[ok])
        allr.append((offsets[si] + np.nonzero(ok)[0]).astype(np.int32))
    keys, rows = np.concatenate(allk), np.concatenate(allr)
    o = np.argsort(keys, kind='stable')
    np.save(os.path.join(a.out, 'keys.npy'), keys[o])
    np.save(os.path.join(a.out, 'rows.npy'), rows[o])
    u = len(np.unique(keys))
    json.dump({'n': int(len(keys)), 'unique_keys': int(u), 'failed': nfail, 'shards': [s['name'] for s in shards],
               'source_idx': os.path.abspath(a.idx), 'key': 'InChIKey first block, blake2b-64'},
              open(os.path.join(a.out, 'info.json'), 'w'), indent=1)
    print(f'merged: {len(keys)} keys ({u} distinct = {len(keys) / max(u, 1):.2f} rows per distinct skeleton), '
          f'{nfail} failed; total {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
