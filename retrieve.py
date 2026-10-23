"""
Retrieval of PURCHASABLE molecules for a reference ligand and a pocket (README 2.24).

  python retrieve.py --smiles "<reference>" --complex Complex/6h0f.pdb --pocket-sdf Complex/6h0f.sdf --pocket-lig Y70_B_502 \
         --anchor-frag 1 --k 30000 --pocket-top 300 --workers 8 --out retrieve_6h0f.csv

0. Anchor-first alternative: `--hits hits.parquet` (anchor_scan.py) replaces step 1 and 2's neighbour search.
1. ErG nearest neighbours of the reference in the 131M-molecule ErG index (exact scan, `erg_index.Index.search`, reads the
   42 GB index: minutes; `--shards N` uses only the first N shards for a test).
2. Keep rows flagged purchasable (purchasable_rows.npy & --mask, default 6 = medium + ZINC-only), neutralise (largest
   component), one row per molecule (InChIKey first block), exact ErG Tanimoto of the NEUTRAL form to the reference.
   The index holds charged forms, so the neighbour search itself is approximate for ionisable references: `--k` is large and
   the neutral ErG is recomputed; `--query-smiles` lets the search use a charged form of the reference.
3. Anchor containment (default on): the candidate must contain the anchor of the reference (the most pharmacophore-rich
   block(s), same selection as grow.py: --anchor-n / --anchor-frag), else there is no pose to fix.
4. Pocket scoring of the best `--pocket-top` by ErG: anchor atoms fixed at the crystal coordinates, ETKDG conformers,
   excluded volume, burial, contacts and polar complementarity exactly as in grow.py (`pocket.place`); every distinct anchor
   match (up to --max-matches) is tried, the best feasible pose is kept.
5. Score = ErG Tanimoto + w3d * s3d (same form as grow.py), vendors and bioactivity attached (vendor_map, bio_map).
Without --complex steps 3-4 are skipped except the containment filter (needs only the reference).
"""
import paths
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import erg_index as X
import erg_v2 as E
import known as K
import grow as G

RDLogger.DisableLog('rdApp.*')
_UN = rdMolStandardize.Uncharger()
_PK = None
_ARGS = None


def neutral(smi):
    try:
        m = Chem.MolFromSmiles(max(smi.split('.'), key=len))
        return _UN.uncharge(K.repair_charges(m)) if m is not None else None
    except Exception:
        return None


def anchor_atoms(ref, tgt, a):
    """Reference atom indices of the anchor block(s); the whole reference when it has no BRICS bond."""
    from frag_erg import site_aware_cuts
    if not site_aware_cuts(ref):
        return set(range(ref.GetNumAtoms()))
    inst, _ = G.pick_anchor(ref, tgt, a.anchor_n, a.anchor_frag)
    return {v for i in inst for v in i['ref_atoms'].values()}


def anchor_query(ref, atoms):
    """SMARTS query of the anchor atoms; query atom -> reference atom index through the atom-map numbers."""
    m = Chem.Mol(ref)
    for at in m.GetAtoms():
        at.SetAtomMapNum(at.GetIdx() + 1)
    q = Chem.MolFromSmarts(Chem.MolFragmentToSmarts(m, atomsToUse=sorted(atoms)))
    order = [at.GetAtomMapNum() - 1 for at in q.GetAtoms()]
    for at in q.GetAtoms():
        at.SetAtomMapNum(0)
    return q, order


def _init(pocket, args):
    global _PK, _ARGS
    _PK, _ARGS = pocket, args


def _place(task):
    """task = (smiles, [match tuples as [(cand idx, ref idx)]]) -> best pose dict (without xyz) or None."""
    smi, maps = task
    mol = Chem.MolFromSmiles(smi)
    res = []
    for amap in maps:
        fixed = {i: _PK.ref_xyz[j] for i, j in amap if j in _PK.ref_xyz}
        if len(fixed) < 3:
            continue
        r = _PK.place(mol, fixed, set(fixed), _PK.n_conf)
        if r['xyz'] is not None and _PK.accept < r['clash'] <= _PK.hopeless and _PK.stage2_confs > _PK.n_conf:
            r2 = _PK.place(mol, fixed, set(fixed), _PK.stage2_confs, seed=_PK.seed + 1)
            res.append(r2 if r2['xyz'] is not None and (r2['clash'], r2['overlap']) < (r['clash'], r['overlap']) else r)
        else:
            res.append(r)
    if not res:
        return None
    r = _PK._pick(res)
    r.pop('xyz', None)
    r['n_matches'] = len(maps)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smiles', required=True, help='reference ligand')
    ap.add_argument('--query-smiles', default=None, help='form of the reference used for the index search (default --smiles)')
    ap.add_argument('--pubchem-idx', default=paths.pubchem_idx())
    ap.add_argument('--flags', default=os.path.join(HERE, 'purchasable_rows.npy'))
    ap.add_argument('--mask', type=int, default=6)
    ap.add_argument('--k', type=int, default=30000)
    ap.add_argument('--hits', default=None, help='anchor_scan.py output: use these molecules instead of the ErG neighbour search')
    ap.add_argument('--max-erg', type=int, default=300000, help='with --hits: cap before the ErG ranking')
    ap.add_argument('--shards', type=int, default=None, help='use only the first N index shards (test)')
    ap.add_argument('--anchor-n', type=int, default=1)
    ap.add_argument('--anchor-frag', type=int, default=None)
    ap.add_argument('--no-anchor-filter', action='store_true')
    ap.add_argument('--complex', default=None)
    ap.add_argument('--pocket-sdf', default=None)
    ap.add_argument('--pocket-lig', default=None)
    ap.add_argument('--pocket-radius', type=float, default=14.0)
    ap.add_argument('--pocket-confs', type=int, default=20)
    ap.add_argument('--pocket-max-clash', type=int, default=2)
    ap.add_argument('--pocket-max-unsat', type=int, default=-1)
    ap.add_argument('--pocket-drop-het', default='')
    ap.add_argument('--pocket-top', type=int, default=100)
    ap.add_argument('--max-matches', type=int, default=3, help='anchor matches (poses) tried per candidate')
    ap.add_argument('--w3d', type=float, default=0.15)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--vendors', default=os.path.join(HERE, 'vendor_map'))
    ap.add_argument('--bio', default=os.path.join(HERE, 'bio_map'))
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--out', default='retrieve.csv')
    a = ap.parse_args()
    t0 = time.time()
    ref = Chem.MolFromSmiles(a.smiles)
    ref_raw = E.erg_raw(ref)
    tgt = G.Target(ref_raw)
    ref_key = K.mol_key(ref)

    if a.hits:
        h = pd.read_parquet(a.hits)
        if len(h) > a.max_erg:
            print(f'{len(h)} anchor hits: random sample of {a.max_erg} (--max-erg) goes on to the ErG ranking')
            h = h.sample(a.max_erg, random_state=1)
        rows, ids, smi, sim = h.row.to_numpy(), h.cid.to_numpy(), h.smiles.to_numpy(), np.zeros(len(h), np.float32)
        print(f'{len(h)} purchasable molecules containing the anchor (from {a.hits})', flush=True)
    else:
        ix = X.Index(a.pubchem_idx)
        if a.shards:
            ix.shards = ix.shards[:a.shards]
            ix.offsets = ix.offsets[:a.shards + 1]
            ix.n = int(ix.offsets[-1])
        print(f'ErG index: {ix.n:,} molecules; searching the {a.k} nearest neighbours of the reference ...', flush=True)
        r = ix.search([a.query_smiles or a.smiles], k=a.k, metric='tanimoto', workers=a.workers, log=print)[0]
        print(f'  index search done in {time.time() - t0:.0f}s', flush=True)
        rows = r['global_row']
        flags = np.load(a.flags, mmap_mode='r')
        keep = (flags[rows] & a.mask) > 0
        rows, sim = rows[keep], r['similarity'][keep]
        print(f'{len(r["global_row"])} neighbours, {len(rows)} flagged purchasable (mask {a.mask})')
        ids, smi = ix.fetch(rows)

    best = {}
    for g, i, s, sim_i in zip(rows, ids, smi, sim):
        m = neutral(s)
        if m is None:
            continue
        key = K.mol_key(m)
        iso = any(at.GetIsotope() for at in m.GetAtoms())
        comp = s.count('.') + 1
        c = dict(row=int(g), cid=int(i), stored_smiles=s, smiles=Chem.MolToSmiles(m), index_sim=float(sim_i),
                 components=comp, is_reference=bool(key == ref_key), n_forms=1, _m=m)
        rank = (iso, comp, int(i))             # one row per InChIKey block: the parent form (no isotopes, single component, lowest CID)
        old = best.get(key)
        if old is None:
            best[key] = (rank, c)
        else:
            n = old[1]['n_forms'] + 1
            if rank < old[0]:
                best[key] = (rank, c)
            best[key][1]['n_forms'] = n
    cand = [v[1] for v in best.values()]
    vec = np.array([E.erg_raw(c['_m']) for c in cand])
    et = tgt.tanimoto(vec)
    for c, t in zip(cand, et):
        c['erg_tanimoto'] = float(t)
    print(f'{len(cand)} distinct neutral molecules; ErG Tanimoto to the reference (neutral form): '
          f'max {et.max():.3f}, median {np.median(et):.3f}')

    pocket = None
    amaps = {}
    if a.complex or not a.no_anchor_filter:
        atoms = anchor_atoms(ref, tgt, a)
        q, order = anchor_query(ref, atoms)
        print(f'anchor: {len(atoms)} reference atoms {sorted(atoms)}, SMARTS {Chem.MolToSmarts(q)}')
    n_anchor = 0
    if not a.no_anchor_filter:
        for c in cand:
            ms = c['_m'].GetSubstructMatches(q, uniquify=False, maxMatches=24)
            c['anchor_matches'] = len(ms)
            if ms:
                amaps[c['row']] = [[(m_[k], order[k]) for k in range(len(order))] for m_ in ms]
        n_anchor = len(amaps)
        cand = [c for c in cand if c['row'] in amaps]
        print(f'{len(cand)} contain the anchor')
    cand.sort(key=lambda c: -c['erg_tanimoto'])

    if a.complex and cand:
        import pocket as PK
        pocket = PK.Pocket(a.complex, a.pocket_sdf, a.pocket_lig, a.pocket_radius, a.pocket_confs, accept=a.pocket_max_clash,
                           drop_het=[x for x in a.pocket_drop_het.split(',') if x])
        pocket.set_reference(ref)
        pocket.score_poses = True
        top = cand[:a.pocket_top]
        tasks = []
        for c in top:
            mm = amaps[c['row']]
            seenm, uniq = set(), []
            for amap in mm:
                kk = tuple(sorted(amap))
                if kk not in seenm:
                    seenm.add(kk)
                    uniq.append(amap)
            tasks.append((c['smiles'], uniq[:a.max_matches]))
        print(f'3D check of {len(tasks)} candidates (anchor fixed, {pocket.n_conf} conformers each) ...', flush=True)
        if a.workers > 1:
            with mp.Pool(a.workers, initializer=_init, initargs=(pocket, a)) as pool:
                out = pool.map(_place, tasks, chunksize=1)
        else:
            _init(pocket, a)
            out = [_place(t) for t in tasks]
        for c, o in zip(top, out):
            if o:
                for k, v in o.items():
                    c['pocket_' + k] = v
        cand = top
        ok = [c for c in cand if c.get('pocket_clash', 99) <= a.pocket_max_clash and
              (a.pocket_max_unsat < 0 or c.get('pocket_n_unsat', 0) <= a.pocket_max_unsat)]
        print(f'{len(ok)} of {len(cand)} pass the excluded-volume rule (<= {a.pocket_max_clash} clashing atoms)')
        cand = ok
    for c in cand:
        c['score'] = c['erg_tanimoto'] + (a.w3d * c.get('pocket_s3d', 0.0) if pocket else 0.0)
    cand.sort(key=lambda c: -c['score'])
    df = pd.DataFrame([{k: v for k, v in c.items() if k != '_m'} for c in cand])

    if len(df):
        if os.path.exists(a.vendors + '_cid.npy'):
            import vendor_annotate as V
            ann = V.VendorMap(a.vendors).annotate(df.cid.tolist())
            df['n_vendors'] = [ann[c][0] for c in df.cid]
            df['n_tight'] = [ann[c][3] for c in df.cid]
            df['vendors'] = [ann[c][1] for c in df.cid]
            df['sid_url'] = [f'https://pubchem.ncbi.nlm.nih.gov/substance/{ann[c][2]}' if ann[c][2] else '' for c in df.cid]
        if os.path.exists(a.bio + '_summary.parquet'):
            import bio_annotate as B
            ba = B.BioMap(a.bio, df.cid.tolist()).annotate(df.cid.tolist())
            for col in ('n_tested', 'n_active', 'n_gene_active', 'best_nM', 'best_gene'):
                df['bio_' + col] = df.cid.map(ba[col])
    df.to_csv(a.out, index=False)
    pd.set_option('display.width', 250, 'display.max_colwidth', 60)
    cols = [c for c in ('cid', 'score', 'erg_tanimoto', 'pocket_clash', 'pocket_s3d', 'n_vendors', 'bio_n_active', 'bio_n_tested',
                        'bio_best_nM', 'is_reference', 'smiles') if c in df]
    print(f'\n{len(df)} retrieved purchasable molecules, {time.time() - t0:.0f}s\n')
    if len(df):
        print(df.head(a.top)[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nwritten {a.out}')


if __name__ == '__main__':
    main()
