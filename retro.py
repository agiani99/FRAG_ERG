"""
Retrospective test on ChEMBL: does ErG-guided growth around an anchor move towards the known actives of the
target, more than a control that has the same anchor, constraints and filters but no (or wrong) guidance?

per target (>= MIN_ACT actives with pChEMBL >= 7):
    reference  = the most potent compound (22-40 heavy atoms, >= 2 rings, >= 3 site-aware BRICS cuts), neutralised
    actives A  = the other compounds with pChEMBL >= 7 (held out; never seen by the grower)
    weak W     = compounds with pChEMBL <= 5 (specificity check)
    G  guided  : target = ErG of the reference
    X  wrong   : target = ErG of a random ChEMBL molecule from elsewhere; anchor/constraints/filters from the reference
    R  random  : expansions, beam and caps chosen at random; anchor/constraints/filters from the reference
  top 10 molecules per condition (G, X: best ErG Tanimoto to their own target; R: random), then
    simA   mean over molecules of max ECFP4 Tanimoto to A
    simW   same against W
    blkA   fraction of the grown (non-anchor) blocks that also occur among the BRICS blocks of A
    sim_ref  mean ECFP4 to the reference; ref_found  reference itself rediscovered
  Background: the same simA for 200 random ChEMBL molecules of similar size.

usage: python retro.py [n_targets=30] [workers=12] [--min-act 80] [--beam 40] [--steps 6] [--seed 0]
"""
import paths
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, argparse, time, multiprocessing as mp
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

import frag_erg as F
import erg_v2 as E
import grow as G

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
CHEMBL = paths.data('ALL_chembl_ErG.parquet')
_UNCH = rdMolStandardize.Uncharger()
_W = {}


def neutral(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = _UNCH.uncharge(rdMolStandardize.LargestFragmentChooser().choose(m))
    except Exception:
        return None
    return m


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def max_sim(fps, ref_fps):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)) for f in fps])


def init(lib_args, args):
    _W['lib'] = G.load_library(*lib_args)
    _W['a'] = args


def one_target(job):
    tid, name, ref_smi, act_smis, weak_smis, pool_smis, seed = job
    t0 = time.time()
    a, lib = _W['a'], _W['lib']
    rng = np.random.default_rng(seed)
    A = [m for m in (neutral(s) for s in act_smis) if m is not None]
    W = [m for m in (neutral(s) for s in weak_smis) if m is not None]
    fA, fW = [fp(m) for m in A], [fp(m) for m in W]
    blocksA = set()
    for m in A:
        try:
            blocksA.update(k for k in F.fragment_keys(m) if '*' in k)
        except Exception:
            pass
    row = dict(target=tid, name=name, ref=ref_smi, n_act=len(A), n_weak=len(W))
    bg = [fp(m) for m in (Chem.MolFromSmiles(s) for s in rng.choice(pool_smis, 50, replace=False))]
    row['bg_simA'] = float(max_sim(bg, fA).mean())

    def score(cond, df, pick):
        if df is None or len(df) < 3:
            row[f'{cond}_n'] = 0 if df is None else len(df)
            return
        top = df.head(10) if pick == 'top' else df.sample(min(10, len(df)), random_state=int(seed))
        ms = [Chem.MolFromSmiles(s) for s in top.smiles]
        fps = [fp(m) for m in ms]
        row[f'{cond}_n'] = len(df)
        row[f'{cond}_simA'] = float(max_sim(fps, fA).mean())
        row[f'{cond}_simW'] = float(max_sim(fps, fW).mean()) if fW else np.nan
        blk = []
        for kk in top.block_keys:
            ks = [k for k in kk.split('|') if not k.startswith('anchor')]
            if ks:
                blk.append(np.mean([k in blocksA for k in ks]))
        row[f'{cond}_blkA'] = float(np.mean(blk)) if blk else np.nan
        row[f'{cond}_sim_ref'] = float(top.ecfp4_to_ref.mean())
        row[f'{cond}_erg_ref'] = float(top.erg_to_ref.mean())

    # G: guided
    dfG, infoG = G.run_grow(a, lib, ref_smi, verbose=False)
    row['G_ref_found'] = infoG['stat']['ref_found']
    row['anchor'] = ' '.join(infoG['anchor_keys'])
    score('G', dfG, 'top')
    # X: wrong target
    x_smi = pool_smis[int(rng.integers(len(pool_smis)))]
    dfX, _ = G.run_grow(a, lib, ref_smi, target_smiles=x_smi, verbose=False)
    score('X', dfX, 'top')
    # R: random growth
    dfR, _ = G.run_grow(a, lib, ref_smi, rng=np.random.default_rng(seed + 1), verbose=False)
    score('R', dfR, 'random')
    row['seconds'] = time.time() - t0
    return row


def pick_targets(df, n, min_act, seed):
    """df: pChEMBL-annotated rows. Returns jobs for n random qualifying targets."""
    rng = np.random.default_rng(seed)
    act = df[df.pchembl_value >= 7].drop_duplicates(['chembl_target_id', 'molecule_chembl_id'])
    cnt = act.groupby('chembl_target_id').size()
    tids = cnt[cnt >= min_act].index.to_numpy()
    rng.shuffle(tids)
    pool = df.drop_duplicates('molecule_chembl_id').SMILES
    pool = pool[(pool.str.len() > 25) & (pool.str.len() < 90)].sample(3000, random_state=seed).tolist()
    pool = [Chem.MolToSmiles(m) for m in (neutral(s) for s in pool) if m is not None
            and 25 <= m.GetNumHeavyAtoms() <= 40]
    jobs = []
    for tid in tids:
        sub = df[df.chembl_target_id == tid].drop_duplicates('molecule_chembl_id')
        best = sub.sort_values('pchembl_value', ascending=False)
        ref = None
        for _, r in best.head(40).iterrows():
            m = neutral(r.SMILES)
            if m is None or not (22 <= m.GetNumHeavyAtoms() <= 40) or m.GetRingInfo().NumRings() < 2:
                continue
            if set(a.GetSymbol() for a in m.GetAtoms()) - set('C N O S F Cl Br'.split()):
                continue
            try:
                if len(F.site_aware_cuts(m)) < 3:
                    continue
            except Exception:
                continue
            ref = (Chem.MolToSmiles(m), r.molecule_chembl_id)
            break
        if ref is None:
            continue
        acts = sub[(sub.pchembl_value >= 7) & (sub.molecule_chembl_id != ref[1])].SMILES
        weak = sub[sub.pchembl_value <= 5].SMILES
        jobs.append((tid, sub.chembl_target_name.iloc[0], ref[0],
                     acts.sample(min(300, len(acts)), random_state=seed).tolist(),
                     weak.sample(min(300, len(weak)), random_state=seed).tolist() if len(weak) else [],
                     pool, int(rng.integers(1 << 30))))
        if len(jobs) >= n:
            break
    return jobs


def summarize(res):
    from scipy.stats import wilcoxon
    out = []
    ok = res.dropna(subset=['G_simA', 'X_simA', 'R_simA'])
    out.append(f'targets with all three conditions: {len(ok)} of {len(res)}')
    for metric, label in (('simA', 'max ECFP4 to held-out actives'), ('blkA', 'grown blocks seen in actives'),
                          ('erg_ref', 'ErG Tanimoto to the reference')):
        o = res.dropna(subset=[f'G_{metric}', f'X_{metric}', f'R_{metric}'])
        out.append(f'\n{label}  (n={len(o)} targets)')
        out.append(f'   median  guided {o[f"G_{metric}"].median():.3f}   wrong-target {o[f"X_{metric}"].median():.3f}'
                   f'   random {o[f"R_{metric}"].median():.3f}')
        for c, nm in (('X', 'wrong-target'), ('R', 'random')):
            d = o[f'G_{metric}'] - o[f'{c}_{metric}']
            p = wilcoxon(d).pvalue if (d != 0).any() else float('nan')
            out.append(f'   guided - {nm}: median diff {d.median():+.3f}, guided higher in {(d > 0).sum()}/{len(d)}, '
                       f'Wilcoxon p = {p:.3g}')
    o = res.dropna(subset=['G_simA', 'G_simW'])
    out.append(f'\nspecificity: guided molecules, ECFP4 to actives {o.G_simA.median():.3f} vs to weak compounds '
               f'{o.G_simW.median():.3f}  (n={len(o)})')
    out.append(f'background (50 random ChEMBL molecules of similar size) max ECFP4 to actives: '
               f'median {res.bg_simA.median():.3f}')
    out.append(f'reference rediscovered: {int(res.G_ref_found.fillna(False).sum())}/{len(res)} targets (guided)')
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('n_targets', nargs='?', type=int, default=30)
    ap.add_argument('workers', nargs='?', type=int, default=12)
    ap.add_argument('--min-act', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--beam', type=int, default=40)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--anchor-n', type=int, default=2)
    ap.add_argument('--close-top', type=int, default=40)
    ap.add_argument('--out', default=os.path.join(HERE, 'retro_results.csv'))
    r = ap.parse_args()

    a = G.build_parser().parse_args(['--smiles', 'x', '--beam', str(r.beam), '--steps', str(r.steps),
                                     '--anchor-n', str(r.anchor_n),
                                     '--close-top', str(r.close_top)])
    lib_args = (a.library, a.min_count, a.max_block_heavy, a.max_block_attach, 1, a.allow_enamine, a.allow_motifs)
    G.load_library(a.library, a.min_count, a.max_block_heavy, a.max_block_attach, r.workers,
                   a.allow_enamine, a.allow_motifs)                      # build the cache once
    d = pd.read_parquet(CHEMBL, columns=['chembl_target_id', 'chembl_target_name', 'molecule_chembl_id',
                                         'SMILES', 'pchembl_value']).dropna(subset=['pchembl_value'])
    jobs = pick_targets(d, r.n_targets, r.min_act, r.seed)
    print(f'{len(jobs)} targets; beam {r.beam}, steps {r.steps}, anchor-n {r.anchor_n}', flush=True)
    rows, t0 = [], time.time()
    with mp.Pool(r.workers, initializer=init, initargs=(lib_args, a)) as pool:
        for k, row in enumerate(pool.imap_unordered(one_target, jobs), 1):
            rows.append(row)
            pd.DataFrame(rows).to_csv(r.out, index=False)
            print(f'[{k}/{len(jobs)}] {row["target"]} {row["name"][:40]}: '
                  f'simA G {row.get("G_simA", float("nan")):.3f} X {row.get("X_simA", float("nan")):.3f} '
                  f'R {row.get("R_simA", float("nan")):.3f}  ({row["seconds"]:.0f}s, {time.time() - t0:.0f}s total)',
                  flush=True)
    res = pd.DataFrame(rows)
    # background: random ChEMBL molecules of similar size vs the actives of each target
    print('\n' + summarize(res))
    print(f'\nper-target results: {r.out}')
