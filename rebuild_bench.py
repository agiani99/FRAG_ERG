"""
Rebuild-known-molecules benchmark: can the grower regenerate a molecule that is known to exist?

Test molecules: approved / clinical drugs (ERG_SWITCHER/data/MAX_FDAapproved, MAX_clinical) and the most potent ChEMBL
compound of each retro target (retro_results.csv): neutralised, 20-40 heavy atoms, >= 2 rings, >= 3 site-aware BRICS bonds,
elements C N O S F Cl Br. For a test molecule M the grower gets M as the reference (target = ErG of M); success = the search
reaches M itself (same heavy-atom graph; `ref_found`).

Conditions (same library, filters and settings as a normal run, no pocket):
  n1         the most influential block is kept as the anchor, everything else must be grown
  n2         the two most influential connected blocks are kept
  leaf       everything except ONE terminal block (the largest one; --leaf-mode random for a random one) is kept: the
             grower has to find the missing piece (a missing CH3 or F would be trivial, hence the largest)
  leaf_rand  as 'leaf' but with random choices instead of ErG guidance (control: what chance alone gives)

Ceiling control: M can only be rebuilt if the blocks that have to be grown exist in the library (PubChem blocks seen >= 3
times, <= 20 heavy atoms, filters), so every run records `buildable`; recovery is reported overall and among buildable runs.

Random-guidance controls for the anchor conditions (added 2026-10-19): n1_rand / n2_rand = same anchor as n1 / n2 but random
choices (like leaf_rand for leaf): the chance level at the library in use.

Restricted library (added 2026-10-19; --library-mode restricted): the library is reduced to the blocks of the test molecule
itself (as in the oracle library) plus N distractor blocks drawn from the standard library (--n-distractors 0,100,1000,10000;
--reps replicates of the draw for N > 0; --distractor-weight uniform|count). It measures how recovery from one or two blocks
depends on the size of the candidate pool: the oracle library (full library + own blocks) is the right-hand end of the curve.
Distractors never include a block of the test molecule (compared on the stereo-free normalised key). Not part of 'both'.

--no-closable-check (added 2026-10-19): switches off grow.closable() in the workers. The restricted library holds few mono-valent
caps, so with the rule on, states with an open linker end are dropped and recovery at N = 0 is far below what the pool allows
(README 2.33: 28 / 51 % with the rule, 76 / 87 % without, n1 / n2). Use it for the restricted sweep; the full-library modes keep the rule.

usage: python rebuild_bench.py [n_molecules=100] [workers=12] [--seed 0] [--beam 40] [--steps 6] [--conds n1,n2,leaf,leaf_rand]
       [--tag rebuild] [--no-chembl] [--library-mode std|oracle|both|restricted] [--n-distractors 0,100,1000,10000] [--reps 3]
"""
import paths
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, argparse, time, multiprocessing as mp
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

import frag_erg as F
import erg_v2 as E
import grow as G
import known as K

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
# drug sets: paths.data('MAX_FDAapproved.parquet') / paths.data('MAX_clinical.parquet') (FRAGERG_DATA)
ELEMENTS = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br'}
_UN, _LF = rdMolStandardize.Uncharger(), rdMolStandardize.LargestFragmentChooser()
_W = {}


def neutral(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return _UN.uncharge(K.repair_charges(_LF.choose(m)))
    except Exception:
        return None


def norm_key(k):
    """Stereo-free canonical form of a dummy-SMILES block key."""
    m = Chem.MolFromSmiles(k)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else k


def structure(m):
    """Blocks of m as pick_anchor numbers them: per-block normalised keys, heavy atoms, and the leaf blocks."""
    cuts = F.site_aware_cuts(m)
    cm, dums = F.cut_mol(m, cuts)
    maps = []
    frs = list(rdmolops.GetMolFrags(cm, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=maps))
    fid = {a: k for k, mp_ in enumerate(maps) for a in mp_}
    adj = {k: set() for k in range(len(frs))}
    for di, dj in dums:
        adj[fid[di]].add(fid[dj])
        adj[fid[dj]].add(fid[di])
    keys, heavy = [], []
    for fr in frs:
        try:
            keys.append(norm_key(Chem.MolToSmiles(fr)))
        except Exception:
            keys.append('?')
        heavy.append(sum(1 for a in fr.GetAtoms() if a.GetAtomicNum() != 0))
    leaves = [k for k in adj if len(adj[k]) == 1]
    return dict(keys=keys, heavy=heavy, leaves=leaves, n=len(frs))


def select_molecules(n, seed, use_chembl):
    rows = []
    for f in ('MAX_FDAapproved', 'MAX_clinical'):
        d = pd.read_parquet(paths.data(f'{f}.parquet'), columns=['Name', 'SMILES'])
        rows += [('drug:' + str(nm), s) for nm, s in zip(d.Name, d.SMILES)]
    if use_chembl and os.path.exists(os.path.join(HERE, 'retro_results.csv')):
        r = pd.read_csv(os.path.join(HERE, 'retro_results.csv'))
        rows += [('chembl:' + str(t), s) for t, s in zip(r.target, r.ref)]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    out, seen = [], set()
    for name, smi in rows:
        m = neutral(smi)
        if m is None or not (20 <= m.GetNumHeavyAtoms() <= 40) or m.GetRingInfo().NumRings() < 2:
            continue
        if {a.GetSymbol() for a in m.GetAtoms()} - ELEMENTS:
            continue
        try:
            if len(F.site_aware_cuts(m)) < 3:
                continue
        except Exception:
            continue
        key = K.mol_key(m)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, Chem.MolToSmiles(m)))
        if len(out) >= n:
            break
    return out


def block_entries(keys, start_kid):
    """Library entries for the given dummy-SMILES blocks, built exactly like grow._lib_worker but WITHOUT the library
    limits and filters (heavy atoms, attachments, motifs...): the blocks of the test molecule itself."""
    rows = []
    for off, key in enumerate(keys):
        m = Chem.MolFromSmiles(key)
        if m is None:
            continue
        dums = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
        if not dums:
            continue
        t = F.fragment_table(m)
        heavy = m.GetNumAtoms() - len(dums)
        nring = m.GetRingInfo().NumRings()
        for a in dums:
            d = a.GetIdx()
            bt = m.GetBondBetweenAtoms(d, t['attach'][d]).GetBondType()
            p = t['prof'][d]
            p16 = p[:, :G.PW].copy()
            p16[:, G.PW - 1] += p[:, G.PW:].sum(1)
            rows.append((start_kid + off, d, a.GetIsotope() // 10, a.GetIsotope() % 10, int(t['attach_c'][d]),
                         int(t['hyok'][d]), 2 if bt == Chem.BondType.DOUBLE else 1, len(dums), heavy, 1,
                         t['counts'], t['internal'], p16, nring))
    return rows


def augment(lib, keys):
    """Copy of the library with the blocks `keys` added (the oracle library of one test molecule)."""
    keys = [k for k in dict.fromkeys(keys) if '*' in k]
    rows = block_entries(keys, len(lib['keys']))
    if not rows:
        return lib
    new = dict(lib)
    new['keys'] = lib['keys'] + keys
    for name, i in (('kid', 0), ('dum', 1), ('lab', 2), ('dC', 3), ('ac', 4), ('hy', 5), ('bt', 6), ('ndum', 7),
                    ('heavy', 8), ('cnt', 9), ('nring', 13)):
        new[name] = np.concatenate([lib[name], np.array([r[i] for r in rows])])
    new['counts'] = np.concatenate([lib['counts'], np.array([r[10] for r in rows], dtype=np.int16)])
    new['internal'] = np.concatenate([lib['internal'], np.array([r[11] for r in rows], dtype=np.int16)])
    new['prof'] = np.concatenate([lib['prof'], np.array([r[12] for r in rows], dtype=np.float32)])
    return new


_ROW_FIELDS = ('kid', 'dum', 'lab', 'dC', 'ac', 'hy', 'bt', 'ndum', 'heavy', 'cnt', 'counts', 'internal', 'prof', 'nring')


def restrict(lib, kid_norm, own_norm, n, seed, weight='uniform'):
    """Library reduced to n distractor blocks drawn from `lib` (distinct blocks, excluding the blocks of the test molecule:
    `own_norm` = set of their normalised keys; `kid_norm` = {kid: normalised key}). Rows of unselected blocks are dropped;
    `keys` is kept (rows address it through `kid`), so the result has the same layout as the original library."""
    kids = np.array([k for k, nk in kid_norm.items() if nk not in own_norm], dtype=np.int64)
    n = min(int(n), len(kids))
    rng = np.random.default_rng(seed)
    if n <= 0:
        chosen = np.empty(0, dtype=np.int64)
    elif weight == 'count':
        first = {}
        for i, k in enumerate(lib['kid']):
            first.setdefault(int(k), i)
        p = np.array([lib['cnt'][first[int(k)]] for k in kids], dtype=float)
        chosen = rng.choice(kids, size=n, replace=False, p=p / p.sum())
    else:
        chosen = rng.choice(kids, size=n, replace=False)
    mask = np.isin(lib['kid'], chosen)
    new = dict(lib)
    for f in _ROW_FIELDS:
        new[f] = lib[f][mask]
    return new


def init(lib_args, args):
    _W['lib'] = G.load_library(*lib_args)
    _W['a'] = args
    if getattr(args, 'no_closable_check', False):
        # grow.closable() drops every state whose open ends cannot all be capped by a mono-valent library block; a restricted
        # library has few caps, so the rule itself limits recovery there (README 2.33). Switched off in this worker only.
        G.closable = lambda cx, st: True


def one_run(task):
    t0 = time.time()
    a, lib = _W['a'], _W['lib']
    a.anchor_n, a.anchor_keep, a.anchor_frag = task['anchor_n'], task['keep'], None
    from collections import Counter
    a.trace_keys = dict(Counter(task['frag_keys']))
    rng = np.random.default_rng(task['seed']) if task['cond'] in ('leaf_rand', 'n1_rand', 'n2_rand') else None
    if task['lib'] == 'oracle':
        lib = augment(lib, task['own_keys'])
    elif task['lib'] == 'restricted':
        lib = augment(restrict(lib, _W['kid_norm'], set(task['frag_keys']), task['n_dist'], task['dseed'],
                               task['dweight']), task['own_keys'])
    m = Chem.MolFromSmiles(task['smiles'])
    row = {k: task[k] for k in ('id', 'smiles', 'cond', 'lib', 'heavy', 'n_blocks', 'leaf_heavy')}
    row['closable_check'] = not getattr(a, 'no_closable_check', False)
    row['n_dist'] = task.get('n_dist', -1)                      # restricted library: number of distractor blocks (-1 = n/a)
    row['rep'] = task.get('rep', 0)
    try:
        df, info = G.run_grow(a, lib, task['smiles'], rng=rng, verbose=False)
    except Exception as e:                                           # keep the benchmark going; record the failure
        row.update(error=repr(e)[:200], ref_found=False, n_accepted=0, seconds=time.time() - t0)
        return row
    kept = sorted(int(k.replace('anchor', '')) for k in info['anchor_keys'] if k.startswith('anchor'))
    grown = [i for i in range(task['n_blocks']) if i not in kept]
    row['kept'] = ','.join(map(str, kept))
    row['grown_blocks'] = len(grown)
    row['buildable'] = all(task['frag_keys'][i] in _W['libset'] for i in grown)     # w.r.t. the standard library
    if task['lib'] in ('oracle', 'restricted'):
        row['buildable_std'] = row['buildable']
        row['buildable'] = True                         # by construction
    row['ref_found'] = bool(info['stat']['ref_found'])
    row['n_accepted'] = len(df)
    tr = info['stat'].get('trace', [])
    if tr:
        dead = [t['step'] for t in tr if t['beam_on_path'] == 0]
        row['path_lost_at_step'] = dead[0] if dead else 0                  # 0 = an on-path state stayed in the beam
        row['path_steps'] = len(tr)
        row['trace'] = ';'.join(f"{t['step']}:{t['on_path_cands']}/{t['cands']}@{t['best_pos']}>{t['beam_on_path']}/{t['beam']}"
                                for t in tr)
    row['mismatch'] = info['stat']['mismatch']
    if len(df):
        row['best_erg'] = float(df.erg_tanimoto.max())
        row['best_ecfp4_to_M'] = float(df.ecfp4_to_ref.max())
    row['seconds'] = round(time.time() - t0, 1)
    return row


def init2(lib_args, args, libset, kid_norm=None):
    init(lib_args, args)
    _W['libset'] = libset
    _W['kid_norm'] = kid_norm


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(res):
    out = []
    ok = res[res['error'].isna()] if 'error' in res else res
    out.append(f'{res.id.nunique()} molecules, {len(res)} runs ({len(res) - len(ok)} failed)')
    out.append('std = PubChem block library only; oracle = library + the blocks of the test molecule itself '
               '(M rebuildable by construction: tests the search alone)')
    out.append('')
    out.append(f'{"condition":10s}{"library":>8s}{"runs":>6s}{"recovered":>11s}{"rate":>8s}{"95% CI":>16s}'
               f'{"  | std: buildable":>20s}{"recovered among them":>24s}')
    for c in ('n1', 'n2', 'n1_rand', 'n2_rand', 'leaf', 'leaf_rand'):
        for lm in ('std', 'oracle'):
            r = ok[(ok.cond == c) & (ok.lib == lm)]
            if not len(r):
                continue
            k = int(r.ref_found.sum())
            lo, hi = wilson(k, len(r))
            extra = ''
            if lm == 'std':
                nb = int(r.buildable.sum())
                kb = int((r.ref_found & r.buildable).sum())
                extra = f'{nb:>12d}{f"{kb}/{nb}":>24s}'
            out.append(f'{c:10s}{lm:>8s}{len(r):6d}{k:11d}{k / len(r):8.1%}{f"[{lo:.1%}, {hi:.1%}]":>16s}{extra}')
    r = ok[(ok.cond == 'leaf') & (ok.lib == 'oracle')]
    if len(r):
        out.append('')
        out.append('leaf, oracle library: recovery by size of the missing block (heavy atoms):')
        for lo_, hi_ in ((1, 4), (5, 8), (9, 13), (14, 20), (21, 40)):
            s_ = r[(r.leaf_heavy >= lo_) & (r.leaf_heavy <= hi_)]
            if len(s_):
                out.append(f'  {lo_:2d}-{hi_:2d} atoms: {int(s_.ref_found.sum())}/{len(s_)} recovered')
    out.append('')
    out.append('best ECFP4 to the original among the accepted molecules (median over runs that produced candidates):')
    if 'best_ecfp4_to_M' in ok:
        for c in ('n1', 'n2', 'leaf', 'leaf_rand'):
            for lm in ('std', 'oracle'):
                r = ok[(ok.cond == c) & (ok.lib == lm) & ok.best_ecfp4_to_M.notna()]
                if len(r):
                    out.append(f'  {c:10s}{lm:>8s} {r.best_ecfp4_to_M.median():.3f}  (runs with candidates {len(r)}, '
                               f'median accepted {r.n_accepted.median():.0f})')
    if 'path_lost_at_step' in ok:
        out.append('')
        out.append('where M is lost (runs where M was not found; a state is "on path" if its added blocks are a subset of M\'s):')
        for c in ('n1', 'n2', 'leaf'):
            for lm in ('std', 'oracle'):
                r = ok[(ok.cond == c) & (ok.lib == lm) & (~ok.ref_found.astype(bool)) & ok.path_lost_at_step.notna()]
                if len(r):
                    lost = r.path_lost_at_step
                    out.append(f'  {c:6s}{lm:>8s}: {len(r)} failed runs; on-path state still in the beam at the end: '
                               f'{int((lost == 0).sum())}; lost at step 1: {int((lost == 1).sum())}, step 2: {int((lost == 2).sum())}, '
                               f'step 3: {int((lost == 3).sum())}, later: {int((lost >= 4).sum())}')
    rs = ok[ok.lib == 'restricted'] if 'lib' in ok else ok.iloc[0:0]
    if len(rs):
        out.append('')
        out.append('restricted library (the blocks of the test molecule + N distractor blocks from the standard library); '
                   'recovery pooled over the replicate draws of the distractors; 95% Wilson CI over runs')
        out.append(f'{"condition":10s}{"N":>8s}{"runs":>6s}{"recovered":>11s}{"rate":>8s}{"95% CI":>16s}{"median s":>10s}')
        for c in ('n1', 'n1_rand', 'n2', 'n2_rand'):
            for nd in sorted(rs.n_dist.unique()):
                r = rs[(rs.cond == c) & (rs.n_dist == nd)]
                if not len(r):
                    continue
                k = int(r.ref_found.sum())
                lo, hi = wilson(k, len(r))
                out.append(f'{c:10s}{int(nd):8d}{len(r):6d}{k:11d}{k / len(r):8.1%}{f"[{lo:.1%}, {hi:.1%}]":>16s}'
                           f'{r.seconds.median():10.1f}')
        out.append('')
        out.append('same, stratified by the number of blocks to grow (grown_blocks); the reachable fraction falls with it. Note: unless the '
                   'run used --no-closable-check, states whose open ends cannot all be capped by a mono-valent block of the pool are '
                   'dropped, and a small pool has few caps, so low recovery at small N can reflect that rule, not the score '
                   '(README 2.33)')
        for lab_, sel in (('<= 3 blocks to grow', rs.grown_blocks <= 3), ('> 3 blocks to grow', rs.grown_blocks > 3)):
            out.append(f'  [{lab_}]')
            for c in ('n1', 'n1_rand', 'n2', 'n2_rand'):
                for nd in sorted(rs.n_dist.unique()):
                    r = rs[sel & (rs.cond == c) & (rs.n_dist == nd)]
                    if not len(r):
                        continue
                    k = int(r.ref_found.sum())
                    lo, hi = wilson(k, len(r))
                    out.append(f'  {c:10s}{int(nd):8d}{len(r):6d}{k:11d}{k / len(r):8.1%}{f"[{lo:.1%}, {hi:.1%}]":>16s}')
        out.append('guided minus random at each N: compare the guided and *_rand rows; the paired per-molecule comparison is in the CSV '
                   '(columns id, cond, n_dist, rep, ref_found)')
    mm = int(ok['mismatch'].sum()) if 'mismatch' in ok else 0
    out.append('')
    out.append(f'incremental-vs-scratch ErG mismatches over all runs: {mm}')
    return chr(10).join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('n_molecules', nargs='?', type=int, default=100)
    ap.add_argument('workers', nargs='?', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--beam', type=int, default=40)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--per-end', type=int, default=60, help='expansions kept per attachment (grow.py default 60)')
    ap.add_argument('--score', choices=('tanimoto', 'contain'), default='tanimoto')
    ap.add_argument('--overshoot-weight', type=float, default=1.0)
    ap.add_argument('--conds', default='n1,n2,leaf,leaf_rand')
    ap.add_argument('--library-mode', choices=('std', 'oracle', 'both', 'restricted'), default='both',
                    help='std = the PubChem block library only; oracle = the library plus the blocks of the test '
                         'molecule itself (M is then rebuildable by construction: tests the search alone); both = run each')
    ap.add_argument('--n-distractors', default='0,100,1000,10000',
                    help='restricted mode: comma list of numbers of distractor blocks from the standard library')
    ap.add_argument('--reps', type=int, default=3, help='restricted mode: replicate draws of the distractors for N > 0')
    ap.add_argument('--distractor-weight', choices=('uniform', 'count'), default='uniform',
                    help='restricted mode: draw distractors uniformly or in proportion to their PubChem count')
    ap.add_argument('--no-closable-check', action='store_true',
                    help='switch off the grower rule that drops states whose open ends cannot all be capped (use with '
                         '--library-mode restricted, whose pool has few caps; default: rule on, as in all earlier runs)')
    ap.add_argument('--tag', default='rebuild')
    ap.add_argument('--leaf-mode', choices=('largest', 'random'), default='largest',
                    help="which terminal block is removed in 'leaf' / 'leaf_rand' (largest = the informative test)")
    ap.add_argument('--no-chembl', action='store_true')
    r = ap.parse_args()

    a = G.build_parser().parse_args(['--smiles', 'x', '--beam', str(r.beam), '--steps', str(r.steps),
                                     '--per-end', str(r.per_end), '--score', r.score,
                                     '--overshoot-weight', str(r.overshoot_weight)])
    a.no_closable_check = r.no_closable_check
    lib_args = (a.library, a.min_count, a.max_block_heavy, a.max_block_attach, 1, a.allow_enamine, a.allow_motifs)
    lib = G.load_library(a.library, a.min_count, a.max_block_heavy, a.max_block_attach, r.workers,
                         a.allow_enamine, a.allow_motifs)                       # build the cache once
    t0 = time.time()
    libset = {norm_key(k) for k in {lib['keys'][i] for i in np.unique(lib['kid'])}}
    kid_norm = ({int(i): norm_key(lib['keys'][i]) for i in np.unique(lib['kid'])} if r.library_mode == 'restricted' else None)
    print(f'library: {len(libset)} distinct blocks ({time.time() - t0:.0f}s to normalise)', flush=True)
    mols = select_molecules(r.n_molecules, r.seed, not r.no_chembl)
    print(f'{len(mols)} test molecules selected', flush=True)

    rng = np.random.default_rng(r.seed)
    tasks = []
    for k, (name, smi) in enumerate(mols):
        m = Chem.MolFromSmiles(smi)
        st = structure(m)
        leaf = (max(st['leaves'], key=lambda i: st['heavy'][i]) if r.leaf_mode == 'largest'
                else int(rng.choice(st['leaves'])))
        keep_leaf = ','.join(str(i) for i in range(st['n']) if i != leaf)
        base = dict(id=f'{k:03d}|{name}', smiles=smi, heavy=m.GetNumHeavyAtoms(), n_blocks=st['n'],
                    frag_keys=st['keys'], leaf_heavy=st['heavy'][leaf], seed=int(rng.integers(1 << 30)))
        base['own_keys'] = F.fragment_keys(m)
        modes = ('std', 'oracle') if r.library_mode == 'both' else (r.library_mode,)
        for lib_mode in modes:
            if lib_mode == 'restricted':
                variants = [(int(nd), rep_) for nd in r.n_distractors.split(',')
                            for rep_ in range(1 if int(nd) == 0 else r.reps)]
            else:
                variants = [(None, 0)]
            for nd, rep_ in variants:
                extra = {} if nd is None else dict(n_dist=nd, rep=rep_, dweight=r.distractor_weight,
                                                    dseed=int(np.random.default_rng([r.seed, k, nd, rep_]).integers(1 << 30)))
                for cond in r.conds.split(','):
                    if cond in ('n1', 'n1_rand'):
                        tasks.append(dict(base, cond=cond, anchor_n=1, keep=None, lib=lib_mode, **extra))
                    elif cond in ('n2', 'n2_rand'):
                        tasks.append(dict(base, cond=cond, anchor_n=2, keep=None, lib=lib_mode, **extra))
                    elif cond in ('leaf', 'leaf_rand'):
                        tasks.append(dict(base, cond=cond, anchor_n=1, keep=keep_leaf, lib=lib_mode, **extra))
    print(f'{len(tasks)} runs; beam {r.beam}, steps {r.steps}, {r.workers} workers', flush=True)
    rows = []
    out_csv = os.path.join(HERE, f'{r.tag}_results.csv')
    with mp.Pool(r.workers, initializer=init2, initargs=(lib_args, a, libset, kid_norm)) as pool:
        for i, row in enumerate(pool.imap_unordered(one_run, tasks), 1):
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'[{i}/{len(tasks)}] {row["id"][:30]:30s} {row["cond"]:9s} {row.get("lib", ""):6s} found={row.get("ref_found")} '
                  f'buildable={row.get("buildable")} accepted={row.get("n_accepted")} ({row.get("seconds")}s)', flush=True)
    res = pd.DataFrame(rows)
    text = summarize(res)
    open(os.path.join(HERE, f'{r.tag}_summary.txt'), 'w').write(text)
    print('\n' + text)
